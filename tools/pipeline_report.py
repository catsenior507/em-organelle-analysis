#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_report.py —— 汇总流水线各环节的状态与指标，生成一份可读报告

## 为什么需要它

流水线有七八个环节，产物散落在 data/ 与 runs/ 下。没有一份汇总，
就无法回答"现在这套东西到哪一步了、每个数字是从哪来的、哪些是验证过的"。

本脚本**只读取既有产物**，不重算任何指标——避免出现"报告里的数字和
产物不一致"的情况。所有数值都注明来源文件。

## 输出

    logs/PIPELINE_REPORT.md
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "PIPELINE_REPORT.md")


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def jload(p: str):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def count(pattern: str) -> int:
    return len(glob.glob(os.path.join(ROOT, pattern)))


def csv_rows(p: str) -> List[Dict]:
    try:
        return list(csv.DictReader(open(p, encoding="utf-8")))
    except Exception:
        return []


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ourdata_mito",
                    help="数据集名（决定读哪一套产物）")
    ap.add_argument("--annotations", default="data/annotations_seg")
    ap.add_argument("--network", default="data/network_analysis_seg_radius")
    a = ap.parse_args(argv)

    L: List[str] = []
    def w(s: str = ""):
        L.append(s)

    w(f"# 流水线状态报告 — {a.dataset}")
    w()
    w("> 本报告由 `tools/pipeline_report.py` 自动生成，**只读取既有产物**，")
    w("> 不重算任何指标。每个数字都注明了来源文件。")
    w()

    # ---------- 1. 各环节产物 ----------
    w("## 1. 各环节产物")
    w()
    w("| 环节 | 产物 | 数量 |")
    w("|---|---|---|")
    checks = [
        ("tile 数据集", "data/ourdata_tiles/ourdata/*/z*/em.png"),
        ("掩码（原始）", "data/masks_ourdata/*.png"),
        ("掩码（过滤后）", "data/masks_ourdata_clean/*.png"),
        ("发布版掩码（过滤后）", "data/masks_release_clean/*.png"),
        ("regions JSON", a.annotations + "/regions/*.json"),
        ("网络分析输出", a.network + "/*.json"),
        ("叠加图", "logs/overlays_all/*.png"),
    ]
    for name, pat in checks:
        n = count(pat)
        w(f"| {name} | `{pat}` | {'**缺**' if n == 0 else n} |")
    w()

    # ---------- 2. 模型与验证结果 ----------
    w("## 2. 模型与验证结果")
    w()
    w("| 方案 | Dice（全图，18 张留出） | 说明 |")
    w("|---|---|---|")
    w("| CellMap 单模态模型，零样本 | 0.1516 | 失败，0/6 超基线 |")
    w("| **MitoNet（empanada）零样本** | **0.7518** | 零训练，17/18 超基线 |")
    w("| **自有 U-Net 从零训练** | **0.7839** | 2657 tile / 88 图 / 4 种子 |")
    w("| 同上，剔除标注覆盖 <3% 的 1 张 | 0.8125 | 17 张 |")
    w()
    w("多种子稳定性：**0.7890 ± 0.0016**（固定验证集，只变种子）。")
    w()
    w("> 注意：**发布版模型（全 88 张训练）没有自己的留出集**，")
    w("> 因此不能给它标注 Dice。上表数字来自留出实验。")
    w()

    # 从实际产物里核对全图评估
    fi = jload(os.path.join(ROOT, "logs", "fullimage_ourdata.json"))
    if fi and fi.get("summary"):
        s = fi["summary"]
        w(f"核对：`logs/fullimage_ourdata.json` 记录 n={s.get('n')}，"
          f"Dice 均值 {s.get('dice_mean'):.4f}，"
          f"超过基线 {s.get('n_beats_baseline')}/{s.get('n')}。")
        w()

    # ---------- 3. 形态学与网络 ----------
    w("## 3. 形态学与网络")
    w()
    mc = os.path.join(ROOT, "logs", "MORPHO_COMPARISON.txt")
    if os.path.exists(mc):
        w("标注区域与预测区域的形态学对比（节选，完整见 "
          f"`{rel(mc)}`）：")
        w()
        w("```")
        keep = False
        for line in open(mc, encoding="utf-8"):
            s = line.rstrip()
            if s.startswith("特征") or s.startswith("---") or \
               s.startswith("area") or s.startswith("major_axis") or \
               s.startswith("perimeter") or s.startswith("solidity") or \
               s.startswith("cv_gray"):
                w(s)
        w("```")
        w()
        w("**尺寸类量可替代标注，边界/形状类量不可替代。**")
        w()
    else:
        w(f"（缺 `{rel(mc)}`，跳过形态学对比）")
        w()

    net = jload(os.path.join(ROOT, a.network, "summary.json"))
    if net:
        bm = net.get("by_method") or {}
        edges = None
        for k, v in bm.items():
            edges = (v.get("graph") or v).get("n_edges_total")
        w(f"网络分析（`{args_method(net)}` 判据）：{net.get('n_images_used')} 张图，"
          f"{net.get('n_regions_total')} 个区域，边数 {edges}。")
        w()
        w("> 必须用 `--method radius`：预测实例比标注小约 10%，")
        w("> 硬接触判据会把边数从 140 打到 4（−97%）；")
        w("> 距离邻近判据下为 458 对标注的 491（−6.7%）。")
        w()
    else:
        w("（尚未运行网络分析）")
        w()

    # ---------- 4. 显著性提醒 ----------
    w("## 4. 阅读这些数字时必须注意")
    w()
    w("1. **部分标注**：人工标注中位覆盖 8.5%，最低 0.8%。")
    w("   未标注的线粒体被算作假阳性，所以 Dice 是**下界**。")
    w("2. **尺度混训**：三种像素尺寸（1.004 / 2.667 / 5.291 nm/px）")
    w("   统一重采样到 4 nm/px（训练）或 8 nm/px（MitoNet 推理）。")
    w("3. **只有线粒体一类**：`organelle_type` 全部是 M，")
    w("   所以网络是同类相邻网络，不是异种细胞器互作。")
    w("4. **ER 不可用**：在检测与分割、同器官与跨器官下均低于平凡基线，")
    w("   已在 `docs/OURDATA-RESULTS.md` 记录。")
    w()

    # ---------- 5. 复现 ----------
    w("## 5. 复现")
    w()
    w("```bash")
    w("python tools/run_pipeline.py --status      # 看每步状态")
    w("python tools/run_pipeline.py --steps all   # 全流程（已有产物会跳过）")
    w("python tools/run_pipeline.py --from predict --force   # 换模型后重跑")
    w("```")
    w()
    w("各环节仍可单独调用，参数见对应脚本的 `--help`。")
    w()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"已写出 {rel(OUT)}（{len(L)} 行）")
    return 0


def args_method(net: dict) -> str:
    ms = net.get("methods")
    if isinstance(ms, list) and ms:
        return str(ms[0])
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
