#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instance_network.py —— 实例级细胞器互作网络（S-BIAD2983）

## 这篇脚本回答的问题，是本项目论文里明确写着"做不到"的那一条

论文的限制第 4 条：

> 公开的多类数据每类只有一个二值掩码，没有实例编号。因此本文只能回答
> "两类膜是否靠得近"，不能回答"每个高尔基体接触几个内质网"。后者需要实例分割。

S-BIAD2983 的分割是**实例级**的：一个视野里 91 个高尔基体、112 个线粒体、
182 个溶酶体、113 个过氧化物酶体、157 个脂滴。这句话因此可以改写了。

## 性能：为什么用位移法而不是逐实例膨胀

首版对每个 A 实例单独做 3D 膨胀。91 个高尔基体 × 8 个类别对 × (1+null) 次
= 数千次两千万体素的膨胀，单文件十几分钟都出不来。

改为**位移法**：接触等价于"存在偏移 o（|o| ≤ d）使 A 的某标签与 B 的某标签相邻"。
对每个偏移整体平移一次，收集 (A 标签, B 标签) 对，一次遍历得到全部接触关系。

收集到的对列表**可复用**：零模型只需把 A 的标签做双射置换后重算度，
不必再做任何空间运算。

## 零模型

把 A 的实例标签随机置换（保持每个实例的体积谱），重算度分布。
若无差别，说明"谁碰谁"完全由几何决定。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

import numpy as np
import tifffile

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = ("https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/983/S-BIAD2983/Files/"
        "BioImageArchive-mcz/")

CHANNELS = ("golgi", "mito", "lyso", "perox", "LD", "nuc")
PAIRS = [("golgi", "mito"), ("golgi", "lyso"), ("mito", "lyso"),
         ("golgi", "perox"), ("mito", "perox"), ("golgi", "LD"),
         ("mito", "LD"), ("lyso", "LD")]


def fetch_seg(cache, group, stem, ch):
    os.makedirs(cache, exist_ok=True)
    local = os.path.join(cache, f"{stem}-{ch}.tiff")
    if os.path.exists(local):
        return tifffile.imread(local)
    rel = urllib.parse.unquote(f"{group}/Segmented/{stem}-{ch}.tiff")
    url = BASE + urllib.parse.quote(rel)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=900, context=CTX) as r:
            raw = r.read()
    except Exception as e:  # noqa: BLE001
        print(f"    [download failed] {ch}: {type(e).__name__}", flush=True)
        return None
    open(local, "wb").write(raw)
    return tifffile.imread(io.BytesIO(raw))


def offsets(radius):
    out = []
    r2 = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx * dx + dy * dy + dz * dz <= r2:
                    out.append((dx, dy, dz))
    return out


def contact_pairs(A, B, radius):
    a_list, b_list = [], []
    for dx, dy, dz in offsets(radius):
        A2 = np.roll(A, (dx, dy, dz), axis=(0, 1, 2))
        m = (A2 != 0) & (B != 0)
        if not m.any():
            continue
        a_list.append(A2[m])
        b_list.append(B[m])
    if not a_list:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    return np.concatenate(a_list), np.concatenate(b_list)


def degrees(a_lab, b_lab):
    """每个 A 标签碰到多少种 B 标签（排序去重，纯 numpy）。"""
    if a_lab.size == 0:
        return {}
    key = a_lab.astype(np.uint64) * np.uint64(1 << 32) + b_lab.astype(np.uint64)
    uniq = np.unique(key)
    a_of = (uniq >> np.uint64(32)).astype(np.uint32)
    vals, counts = np.unique(a_of, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/biad2983")
    ap.add_argument("--group", default="iNeurons day 7")
    ap.add_argument("--stems", nargs="*", default=[
        "01252024_MSi08L_iN_Day7_BR5_N01_Unmixing_0_cmle.ome"])
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--null", type=int, default=10)
    ap.add_argument("--out", default="logs/INSTANCE_INTERACTION.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(20261115)

    # --group auto：从 logs/biad2983_fields.json 查每个 stem 的分组。
    # 批量跑跨组的视野时必须有这个，否则会去错误的分化天目录下找文件。
    group_of = {}
    if a.group == "auto":
        fld = json.load(open("logs/biad2983_fields.json", encoding="utf-8"))
        group_of = {k: v["group"] for k, v in fld.items()}
        print(f"auto 分组：载入 {len(group_of)} 个视野的分组信息", flush=True)

    results = []
    for stem in a.stems:
        print("\n=== " + stem[:52] + " ===", flush=True)
        segs = {}
        g_use = group_of.get(stem, a.group) if group_of else a.group
        if g_use == "auto":
            print(f"  [跳过] {stem[:40]} 查不到分组", flush=True)
            continue
        for ch in CHANNELS:
            s = fetch_seg(a.cache, g_use, stem, ch)
            if s is not None:
                segs[ch] = s
        if not segs:
            continue
        cell = segs.get("cell")
        if cell is None:
            cell = fetch_seg(a.cache, g_use, stem, "cell")
        if cell is not None:
            print(f"  cell 前景占比 {100*(cell>0).mean():.1f}%", flush=True)

        rec = {"stem": stem,
               "instances": {c: int(len(np.unique(s)) - 1) for c, s in segs.items()},
               "radius": a.radius, "pairs": {}}
        print("  instances: " + "  ".join(f"{c}={rec['instances'][c]}" for c in segs),
              flush=True)

        for x, y in PAIRS:
            if x not in segs or y not in segs:
                continue
            A, B = segs[x], segs[y]
            a_lab, b_lab = contact_pairs(A, B, a.radius)
            if a_lab.size == 0:
                rec["pairs"][f"{x}:{y}"] = {"n_contact_pairs": 0}
                print(f"  {x}:{y:<10} no contact", flush=True)
                continue
            deg = degrees(a_lab, b_lab)
            dv = np.array(list(deg.values()))
            a_ids = np.unique(A)
            a_ids = a_ids[a_ids != 0]

            # 零模型：**受限环面平移**。
            #
            # 病史（两版都不对，都记在这里）：
            #
            # v1 对接触体素列表做标签双射——那是恒等变换：双射只是把每个实例的
            #    体素集合整体改名，度分布作为多重集完全不变，富集恒为 1.00。
            #
            # v2 改成全幅环面平移——仍然错。**细胞只占 796×796×32 体积的 2.9%**
            #    （cell 通道实测 577 962 / 20 311 552 体素），于是随机平移有 97%
            #    的概率把细胞器甩到细胞外的空白区，那里没有任何 B，接触数为 0，
            #    零模型均值被压到 0.00，富集算出来是无穷大。∞ 不是信号，是缺陷。
            #
            # v3（当前）把 A 的质心平移到一个**随机的细胞内位置**，并检查平移后
            #    A 至少有 80% 的体素仍在细胞内。这样零模型才是在"细胞内的可行
            #    位置"里取样，而不是在全空体积里取样。
            nul = []
            if cell is not None:
                cell_pts = np.argwhere(cell > 0)
                A_bool = (A != 0)
                nA = int(A_bool.sum())
                centroid = np.argwhere(A_bool).mean(axis=0) if nA else None
                tries = 0
                while len(nul) < a.null and tries < a.null * 20:
                    tries += 1
                    target = cell_pts[rng.integers(len(cell_pts))]
                    sh = tuple(int(round(target[k] - centroid[k])) for k in range(3))
                    A_s = np.roll(A_bool, sh, axis=(0, 1, 2))
                    inside = float((A_s & (cell > 0)).sum()) / max(1, nA)
                    if inside < 0.80:
                        continue
                    A2 = np.roll(A, sh, axis=(0, 1, 2))
                    x2, y2 = contact_pairs(A2, B, a.radius)
                    d2 = degrees(x2, y2) if x2.size else {}
                    nul.append(float(np.mean(list(d2.values()))) if d2 else 0.0)
            nm = float(np.mean(nul)) if nul else 0.0

            rec["pairs"][f"{x}:{y}"] = {
                "n_contact_voxels": int(a_lab.size),
                "n_a_with_contact": int(len(deg)),
                "n_a_total": int(len(a_ids)),
                "mean_deg": float(dv.mean()),
                "median_deg": float(np.median(dv)),
                "max_deg": int(dv.max()),
                "deg_hist": {int(k): int((dv == k).sum())
                             for k in np.unique(dv)[:20]},
                "null_mean_deg": nm,
                "null_draws_used": len(nul),
                "enrichment": (float(dv.mean()) / nm) if nm > 0 else 0.0,
            }
            print(f"  {x}:{y:<10} contact {len(deg)}/{len(a_ids)} inst  "
                  f"mean {dv.mean():>5.2f} partners  max {dv.max():>3}  "
                  f"null {nm:>5.2f}", flush=True)
        results.append(rec)
        # 每完成一个视野就落盘：20 个视野要跑两小时，中断不该从头再来
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"  [已保存 {len(results)} 个视野]", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
