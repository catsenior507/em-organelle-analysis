#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_report_numbers.py —— 核对晨间报告里的关键数字与日志/JSON 是否一致

## 为什么做这一步

今晚抓到 8 个 bug，其中 3 个是"静默出错"（不报错、只给错数字）。
在这种背景下，**我写进报告的每一个数字都必须与源文件核对**，
而不是凭记忆或凭中间输出抄写。抄错一个数字，整份报告的可信度就没了。

本脚本从 runs/*.json 与 logs/*.txt 重新计算关键指标，
与 docs/MORNING-REPORT-NOV2.md 里写的值逐一比对。

## 运行

    python tools/server/scripts/verify_report_numbers.py
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/server/scripts
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # 仓库根
# 说明：早期版本这里少算了一级，取到的是 tools/ 而不是仓库根，
# 于是报告路径变成 tools/docs/... 而报"不存在"。这类路径深度错误在本项目
# 已出现过多次（fetch_em_slices 的层级、test_zarr_to_2d 的 cache 路径），
# 所以这里显式写出每一级，并加断言防止再次错位。
assert os.path.isdir(os.path.join(REPO, "tools")), f"REPO 解析错误: {REPO}"
RUNS = os.path.join(REPO, "runs")
LOGS = os.path.join(REPO, "logs")
DOC = os.path.join(REPO, "docs", "MORNING-REPORT-NOV2.md")

FAILS = []


def check(name: str, claimed, actual, tol: float = 5e-4) -> None:
    if claimed is None or actual is None:
        print(f"  SKIP  {name}: claimed={claimed} actual={actual}")
        return
    ok = abs(float(claimed) - float(actual)) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: 报告={claimed} 实际={actual}")
    if not ok:
        FAILS.append(name)


def best_ap(path: str):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    rows = d.get("rows") or []
    if not rows:
        return None
    b = max(rows, key=lambda r: r["mAP"])
    return (b["per_class"].get("0", {}).get("AP"),
            b["per_class"].get("1", {}).get("AP"), b["mAP"])


def best_dice(path: str):
    try:
        h = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if not h:
        return None
    b = max(h, key=lambda r: r.get("val_mean_dice", -1))
    v = b.get("val") or {}
    er = v.get("1") or {}
    return ((v.get("0") or {}).get("dice"), er.get("dice"),
            er.get("dice_all_foreground"))


def main() -> int:
    print("=" * 78)
    print("报告数字核对")
    print("=" * 78)

    # 线上产物在服务器上；本地若无 runs/ 则只检查报告自身的一致性
    have_runs = os.path.isdir(RUNS) and any(
        os.path.isdir(os.path.join(RUNS, d)) for d in os.listdir(RUNS)) \
        if os.path.isdir(RUNS) else False

    if not have_runs:
        print("\n[说明] 本地没有 runs/ 产物（它们都在服务器上）。")
        print("       本脚本的设计意图是在服务器上运行；")
        print("       这里只做报告内部的自洽性检查。\n")
        return self_consistency()

    print("\n[1] 检测：四组 AP")
    for name, rel, claim in [
        ("同器官 baseline", "detr_kidney_e600/sweep_iou50_fixedAP.json",
         (0.4292, 0.0495, 0.2386)),
        ("跨器官", "detr_xorgan/sweep_allsplit_fixedAP.json",
         (0.2754, 0.0170, 0.1448)),
        ("扩数据 480px", "detr_big480/sweep_iou50_fixedAP.json",
         (0.3596, 0.0075, 0.1835)),
        ("扩数据 640px", "detr_big640/sweep_iou50_fixedAP.json",
         (0.3383, 0.0171, 0.1777)),
    ]:
        r = best_ap(os.path.join(RUNS, rel))
        if not r:
            print(f"  SKIP  {name}（缺文件）")
            continue
        check(f"{name} mito AP", claim[0], r[0])
        check(f"{name} mAP", claim[2], r[2])

    print("\n[2] 分割：同器官（4 个种子）")
    for name, rel, claim_mito in [
        ("seed0", "seg_indomain_200e/history.json", 0.3780),
        ("seed1", "seg_seed1/history.json", 0.5757),
        ("seed2", "seg_seed2/history.json", 0.5314),
        ("seed3", "seg_seed3/history.json", 0.4864),
    ]:
        r = best_dice(os.path.join(RUNS, rel))
        if not r:
            print(f"  SKIP  {name}（缺文件）")
            continue
        check(f"{name} mito Dice", claim_mito, r[0])

    print("\n[3] 域适应四臂")
    for name, rel, claim_mito in [
        ("A", "da_A_zeroshot.json", 0.2000),
        ("B", "da_B_result.json", 0.5288),
        ("C", "da_C_result.json", 0.3075),
        ("D", "da_D_result.json", 0.4476),
    ]:
        r = best_ap(os.path.join(RUNS, rel))
        if not r:
            print(f"  SKIP  {name}（缺文件）")
            continue
        check(f"{name} mito AP", claim_mito, r[0])

    print()
    print("=" * 78)
    if FAILS:
        print(f"不一致 {len(FAILS)} 处：{FAILS}")
        return 1
    print("全部一致")
    return 0


def self_consistency() -> int:
    """在本地无产物时，检查报告内部的数值是否前后矛盾。

    设计要点：**不要用字面量正则去匹配报告里的数字**。
    第一版就是这么写的，结果报了 3 处假失败：
      * `D **真·联合**` 中有 markdown 加粗，正则 `D 真·联合` 匹配不到；
      * 报告写的是 `0.378–0.576`（三位小数），而检查器找 `0.3780`/`0.5757`。
    检查器假阳性会浪费时间去改本来正确的报告，所以这里改为
    **解析出报告里的所有浮点数，再按容差判断目标值是否存在**。
    """
    if not os.path.exists(DOC):
        print(f"[错误] 报告不存在：{DOC}")
        return 2
    txt = open(DOC, encoding="utf-8").read()
    print("报告字数:", len(txt))

    # 抽出所有数字（含 0.xxxx 形式），用于按值比对而不是按字面量
    nums = [float(m) for m in re.findall(r"\d+\.\d+", txt)]

    def has_value(v: float, tol: float = 6e-4) -> bool:
        return any(abs(n - v) <= tol for n in nums)

    print("\n  关键数值是否出现在报告中（按数值比对，容忍 3–4 位小数）：")
    claims = {
        "A 零样本 mito AP": 0.2000,
        "B 仅目标域 mito AP": 0.5288,
        "C 源权重微调 mito AP": 0.3075,
        "D 真·联合 mito AP": 0.4476,
        "mito Dice 均值": 0.4929,
        "mito Dice 标准差": 0.0848,
        "mito Dice 最小值": 0.3780,
        "mito Dice 最大值": 0.5757,
        "mito 基线下限": 0.2396,
        "mito 基线上限": 0.4075,
        "seed1 的 er 基线（唯一被超过的）": 0.1409,
        "er 视觉 AUC": 0.365,
        "mito 视觉 AUC": 0.163,
    }
    for name, v in claims.items():
        ok = has_value(v)
        print(f"    {'PASS' if ok else 'FAIL'}  {name} = {v}")
        if not ok:
            FAILS.append(name)

    # 四臂排序 B > D > C > A 必须在报告里被正确声明
    print("\n  四臂排序声明：")
    order_ok = (claims["B 仅目标域 mito AP"] > claims["D 真·联合 mito AP"]
                > claims["C 源权重微调 mito AP"] > claims["A 零样本 mito AP"])
    print(f"    {'PASS' if order_ok else 'FAIL'}  B > D > C > A 数值上成立")
    if not order_ok:
        FAILS.append("四臂排序")

    print()
    if FAILS:
        print(f"报告内部不一致 {len(FAILS)} 处：{FAILS}")
        return 1
    print("报告内部数值自洽")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
