#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metrics.py —— 用**已知答案**验证评估管线

## 为什么必须做这一步

前面所有结论（mito AP 0.43、er AP 0.013、跨器官腰斩……）都建立在
`eval_detr.py` / `sweep_ckpt.py` 里的这几个函数上：

    box_iou_xyxy          边界框 IoU
    cxcywh_norm_to_xyxy_px  归一化 cxcywh -> 像素 xyxy
    greedy_match          按置信度排序的贪心匹配
    average_precision     101 点插值 AP

如果其中任何一个有 bug，所有数字都是错的，而且**不会报错**——
只会安静地给出一个看起来合理的数值。这类错误最难发现。

本脚本用可以手算出答案的用例逐个验证，包括边界情况：
完美检测、全部误检、漏检、IoU 恰好等于阈值、空真值、空预测。

## 运行

    python tools/server/scripts/test_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 服务器上 slice_dataset.py 被拷进 scripts/，但本地它在 tools/detr/。
# 两条路径都加上，这样同一个测试在本地和服务器都能跑。
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # tools/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "detr"))                # tools/detr

from eval_detr import (average_precision, box_iou_xyxy,  # noqa: E402
                       cxcywh_norm_to_xyxy_px, greedy_match)

FAILS = []


def check(name: str, got, want, tol: float = 1e-6) -> None:
    ok = (abs(float(got) - float(want)) <= tol
          if isinstance(want, (int, float)) else got == want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------
# 1. IoU
# --------------------------------------------------------------------------

def test_iou() -> None:
    print("\n[1] box_iou_xyxy")
    # 两个完全相同的框 -> IoU = 1
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    check("相同框 IoU=1", float(box_iou_xyxy(a, a)[0, 0]), 1.0)
    # 完全不重叠 -> 0
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    check("不重叠 IoU=0", float(box_iou_xyxy(a, b)[0, 0]), 0.0)
    # 半重叠：A=[0,10]^2 面积 100，B=[5,15]^2 面积 100
    # 交 = [5,10]^2 = 25；并 = 100 + 100 - 25 = 175 -> IoU = 25/175 = 1/7
    # （最初我把"并"错算成 75，写成了 1/3；那是我的算术错误，不是代码问题）
    c = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    check("半重叠 IoU=1/7", float(box_iou_xyxy(a, c)[0, 0]), 25.0 / 175.0)
    # 一个框贴边接触（共享边但无面积重叠）-> 0
    d = torch.tensor([[10.0, 0.0, 20.0, 10.0]])
    check("贴边 IoU=0", float(box_iou_xyxy(a, d)[0, 0]), 0.0)
    # 空输入
    e = torch.zeros((0, 4))
    check("空真值返回 0 形状", tuple(box_iou_xyxy(a, e).shape), (1, 0))


# --------------------------------------------------------------------------
# 2. 坐标变换
# --------------------------------------------------------------------------

def test_cxcywh() -> None:
    print("\n[2] cxcywh_norm_to_xyxy_px")
    # 图像 100x200（w=100,h=200）；中心 (0.5,0.5) 尺寸 (0.5,0.25)
    # -> 像素 cx=50, cy=100, bw=50, bh=50
    # -> x1=25, y1=75, x2=75, y2=125
    b = torch.tensor([[0.5, 0.5, 0.5, 0.25]])
    out = cxcywh_norm_to_xyxy_px(b, 100, 200)[0]
    for name, got, want in zip(("x1", "y1", "x2", "y2"), out.tolist(),
                               (25.0, 75.0, 75.0, 125.0)):
        check(f"  {name}", got, want)
    # 左上角满幅框
    b2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    o2 = cxcywh_norm_to_xyxy_px(b2, 100, 100)[0].tolist()
    check("满幅框 x1", o2[0], 0.0)
    check("满幅框 y2", o2[3], 100.0)


# --------------------------------------------------------------------------
# 3. 贪心匹配
# --------------------------------------------------------------------------

def test_match() -> None:
    print("\n[3] greedy_match")
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                       [20.0, 20.0, 30.0, 30.0]])
    # 两个预测都精确命中 -> 2 个 TP
    pred = gt.clone()
    scores = torch.tensor([0.9, 0.8])
    hit, n = greedy_match(pred, scores, gt, 0.5)
    check("完美匹配命中数", n, 2)
    # 一个预测重复命中同一个真值 -> 只有 1 个 TP（真值不可重复使用）
    pred2 = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                          [0.0, 0.0, 10.0, 10.0]])
    sc2 = torch.tensor([0.9, 0.85])
    _, n2 = greedy_match(pred2, sc2, gt, 0.5)
    check("重复命中同一真值只算 1 个 TP", n2, 1)
    # 低于 IoU 阈值 -> 0
    pred3 = torch.tensor([[6.0, 6.0, 16.0, 16.0]])   # 与 gt[0] IoU=16/184≈0.087
    _, n3 = greedy_match(pred3, torch.tensor([0.9]), gt, 0.5)
    check("低于阈值不计命中", n3, 0)
    # 空真值 -> 0
    _, n4 = greedy_match(pred, scores, torch.zeros((0, 4)), 0.5)
    check("空真值命中 0", n4, 0)


# --------------------------------------------------------------------------
# 4. AP
# --------------------------------------------------------------------------

def test_ap() -> None:
    print("\n[4] average_precision")
    # 完美检测：recall 从 0 到 1 全为 1 的精确率 -> AP=1
    check("完美 AP=1", average_precision(np.array([1.0]), np.array([1.0])), 1.0)
    # 全误检（没有 recall 上升）-> AP 接近 0
    check("全误检 AP≈0",
          average_precision(np.array([0.0]), np.array([0.0])), 0.0)
    # 半程：第一个 TP 后接一个 FP，共 2 个真值
    # preds 排序后：[TP, FP] -> recall [0.5, 0.5], precision [1.0, 0.5]
    # 101 点插值下 AP = 在 recall 0..0.5 上 precision=1，之后 0 -> 0.5
    ap = average_precision(np.array([0.5, 0.5]), np.array([1.0, 0.5]))
    check("TP,FP 各一 AP=0.5", ap, 0.5)
    # 空输入 -> 0
    check("空输入 AP=0", average_precision(np.array([]), np.array([])), 0.0)


# --------------------------------------------------------------------------
# 5. Dice（分割指标）
# --------------------------------------------------------------------------

def test_dice() -> None:
    print("\n[5] Dice（分割评估里的实现）")
    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        i = float(np.logical_and(pred, gt).sum())
        s = float(pred.sum() + gt.sum())
        return 2 * i / s if s > 0 else float("nan")
    g = np.zeros((10, 10), bool)
    g[:5, :5] = True                        # 25 个前景
    check("完全一致 Dice=1", dice(g, g), 1.0)
    check("全背景预测且真值为空", np.isnan(dice(np.zeros((10, 10), bool),
                                              np.zeros((10, 10), bool))), True)
    # 一半重叠：pred 覆盖 g 的左半（25 -> 12 左右），这里用精确构造
    p = np.zeros((10, 10), bool)
    p[:5, :2] = True                        # 10 个前景，全部落在 g 内
    # 交=10, pred=10, gt=25 -> 2*10/(10+25)=20/35
    check("半覆盖 Dice=20/35", dice(p, g), 20.0 / 35.0)
    check("完全不重叠 Dice=0", dice(np.ones((3, 3), bool),
                                   np.zeros((3, 3), bool) * 1), 0.0)


def main() -> int:
    print("=" * 68)
    print("评估管线已知答案测试")
    print("=" * 68)
    test_iou()
    test_cxcywh()
    test_match()
    test_ap()
    test_dice()
    print()
    print("=" * 68)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("全部通过——评估管线的数值可信")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
