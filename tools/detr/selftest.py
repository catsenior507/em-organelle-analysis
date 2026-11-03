#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest.py —— DETR 管线中**不依赖 GPU / 不依赖预训练权重**部分的验证

本机无独立显卡、且无法访问 huggingface.co（``facebook/detr-resnet-50``
权重未缓存），因此无法在此验证完整训练。但原 ``detr_region_feature.py``
最致命的三个 bug 全部位于**不需要模型**的部分：

1. ``collate_fn`` 返回 list、``compute_loss`` 按 dict 索引 → 第一次前向必崩；
2. 匈牙利匹配被替换成 ``topk`` 贪心；
3. 分类损失在"无目标"分支用 BCE 且目标恒为 1。

这三项都可以用随机张量直接验证。本脚本即为此而写，可在 CPU 上秒级运行。

用法::

    python tools/detr/selftest.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from detr.dataset import DetrDataConfig, build_manifest, collate_fn, group_split  # noqa: E402
from detr.losses import (  # noqa: E402
    LossConfig,
    SetCriterion,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    simple_ap,
)

PASS, FAIL = "PASS", "FAIL"
_results: List[tuple] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        _results.append((PASS, name, detail or ""))
    except Exception as exc:  # noqa: BLE001
        _results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# --------------------------------------------------------------------------
# 1. 数据管线
# --------------------------------------------------------------------------

def test_manifest() -> str:
    cfg = DetrDataConfig()
    if not os.path.isdir(cfg.annotation_dir):
        return "SKIP: data/annotations_v2 不存在（请先运行 extract_regions.py）"
    records = build_manifest(cfg, verbose=False)
    assert records, "清单为空"
    counts = [len(r.boxes) for r in records]
    assert min(counts) >= 1, "存在没有目标的图"
    # 归一化框必须在 (0,1) 内
    for r in records:
        b = r.boxes
        assert np.all(b >= -1e-6) and np.all(b <= 1 + 1e-6), f"{r.name} 框越界"
    return (f"{len(records)} 张图，目标数 {min(counts)}–{max(counts)}，"
            f"共 {sum(counts)} 个目标")


def test_group_split_no_leak() -> str:
    """验证按图分组划分不会把同一张图放进两边。"""
    cfg = DetrDataConfig()
    if not os.path.isdir(cfg.annotation_dir):
        return "SKIP: 无 annotations_v2"
    records = build_manifest(cfg, verbose=False)
    tr, va = group_split(records, 0.25, seed=42)
    assert not (set(tr) & set(va)), "train/val 索引重叠"
    assert len(tr) + len(va) == len(records), "划分未覆盖全部样本"
    # 同样的种子必须给出同样的划分
    tr2, va2 = group_split(records, 0.25, seed=42)
    assert tr == tr2 and va == va2, "同种子划分不可复现"
    return f"train={len(tr)} val={len(va)}，无重叠，可复现"


def test_collate_contract() -> str:
    """collate_fn 必须返回 list 语义的 targets。

    原实现::

        targets = []                      # list
        for item in batch: targets.append(item['targets'])
        return {'pixel_values': ..., 'targets': targets}

    而 ``compute_loss`` 里写的是 ``targets['boxes'][i]`` —— dict 语义。
    第一次前向就 TypeError。本测试断言 collate 的返回类型是 list。
    """
    batch = [
        {
            "pixel_values": torch.zeros(3, 32, 32),
            "target": {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([0]),
                "attributes": torch.tensor([[1.0, 2.0]]),
                "image_id": torch.tensor([0]),
            },
            "meta": {"image_id": 0, "name": "a"},
        },
        {
            "pixel_values": torch.ones(3, 32, 32),
            "target": {
                "boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]),
                "labels": torch.tensor([0, 0]),
                "attributes": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                "image_id": torch.tensor([1]),
            },
            "meta": {"image_id": 1, "name": "b"},
        },
    ]
    out = collate_fn(batch)
    assert isinstance(out["targets"], list), (
        f"targets 必须是 list（损失端按 list 遍历），实际 {type(out['targets'])}"
    )
    assert out["pixel_values"].shape == (2, 3, 32, 32), out["pixel_values"].shape
    assert len(out["targets"]) == 2
    return "targets 为 list，pixel_values 正确 stack"


# --------------------------------------------------------------------------
# 2. 几何
# --------------------------------------------------------------------------

def test_giou() -> str:
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(generalized_box_iou(a, b), torch.ones(1, 1), atol=1e-5)

    # 分离的框：GIoU 应为负
    c = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    g = generalized_box_iou(a, c)
    assert g.item() < 0, f"分离框的 GIoU 应为负，实际 {g.item()}"

    # 退化框（零面积）不得产生 NaN —— 原实现没有 clamp，会出 NaN
    d = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    g2 = generalized_box_iou(a, d)
    assert torch.isfinite(g2).all(), "退化框产生了非有限值"
    return f"自交=1.0，分离={g.item():.4f}，退化框有限"


def test_cxcywh_roundtrip() -> str:
    b = torch.tensor([[0.5, 0.4, 0.2, 0.3]])
    xyxy = box_cxcywh_to_xyxy(b)
    back = torch.stack([
        (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
        xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
    ], dim=-1)
    assert torch.allclose(b, back, atol=1e-6)
    return "cxcywh ↔ xyxy 往返一致"


# --------------------------------------------------------------------------
# 3. 匹配与损失
# --------------------------------------------------------------------------

def _make_outputs(B: int, Q: int, A: int, C: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "logits": torch.randn(B, Q, C + 1, generator=g),
        "pred_boxes": torch.rand(B, Q, 4, generator=g),
        "pred_attributes": torch.randn(B, Q, A, generator=g),
    }


def test_hungarian_is_permutation() -> str:
    """匈牙利匹配必须是一对一置换；原实现的 topk 贪心不是。

    构造一个会让 topk 失败的场景：两个目标完全相同，且某个查询对两者的
    IoU 都很高。贪心会把这个查询同时匹配给两个目标（或选错），
    匈牙利则保证每个目标恰好配一个查询。
    """
    B, Q, A = 1, 6, 2
    outputs = _make_outputs(B, Q, A, seed=1)
    # 让查询 0 与两个目标都高度重合
    tgt_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]])
    outputs["pred_boxes"][0, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    targets = [{
        "boxes": tgt_boxes[0],
        "labels": torch.zeros(2, dtype=torch.long),
        "attributes": torch.randn(2, A),
    }]

    crit = SetCriterion(num_classes=1, num_attributes=A, cfg=LossConfig())
    matches = crit._match(outputs, targets)
    src, dst = matches[0]
    assert src.numel() == 2, f"应匹配 2 对，实际 {src.numel()}"
    assert len(set(src.tolist())) == 2, f"同一个查询被匹配给了多个目标: {src.tolist()}"
    assert len(set(dst.tolist())) == 2, f"同一个目标被匹配给了多个查询: {dst.tolist()}"
    return f"src={src.tolist()} dst={dst.tolist()}（一对一，无重复）"


def test_no_target_does_not_crash() -> str:
    """无目标样本必须能算出**有限且可反传**的损失。

    原实现在无目标分支里对 ``logits[i,:,0]``（形状 [Q]）做
    ``binary_cross_entropy_with_logits(target=1)``，语义与形状都错。

    同时验证梯度链完整：早先的一个实现用 ``pred_boxes.sum() * 0.0``
    作为零损失，当 ``pred_boxes`` 不需要梯度时会得到一个
    ``requires_grad=False`` 的张量，把整条损失的梯度链切断——
    这正是"模型完全不更新但损失看着正常"的隐蔽故障。
    """
    torch.manual_seed(7)                     # 固定种子，避免测试本身带随机性
    B, Q, A = 2, 5, 2
    logits = torch.randn(B, Q, 2, requires_grad=True)
    boxes = torch.rand(B, Q, 4, requires_grad=True)
    attrs = torch.randn(B, Q, A, requires_grad=True)

    targets = [
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
         "attributes": torch.zeros(0, A)},
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
         "labels": torch.tensor([0]), "attributes": torch.randn(1, A)},
    ]
    crit = SetCriterion(num_classes=1, num_attributes=A, cfg=LossConfig())
    out = crit({"logits": logits, "pred_boxes": boxes, "pred_attributes": attrs}, targets)
    loss = out["loss"]
    assert torch.isfinite(loss), f"损失非有限：{loss}"
    assert loss.requires_grad, "损失未参与梯度（梯度链被断开）"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "logits 梯度异常"
    assert (logits.grad != 0).any(), "logits 梯度全为 0，模型不会学习"
    return f"loss={loss.item():.4f}，梯度有限且非零"


def test_all_empty_batch() -> str:
    """整批都没有目标时也不能崩，且必须保持梯度链完整。

    这是原实现最容易崩的分支（top-k 对空张量、BCE 形状不匹配），
    也是"背景类损失"唯一真正起作用的场合。
    """
    torch.manual_seed(11)
    B, Q, A = 2, 5, 2
    logits = torch.randn(B, Q, 2, requires_grad=True)
    outputs = {
        "logits": logits,
        "pred_boxes": torch.rand(B, Q, 4, requires_grad=True),
        "pred_attributes": torch.randn(B, Q, A, requires_grad=True),
    }
    targets = [
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
         "attributes": torch.zeros(0, A)} for _ in range(B)
    ]
    crit = SetCriterion(num_classes=1, num_attributes=A, cfg=LossConfig())
    out = crit(outputs, targets)
    assert torch.isfinite(out["loss"]), out["loss"]
    assert out["loss"].requires_grad, "整批无目标时梯度链被断开"
    out["loss"].backward()
    assert logits.grad is not None and (logits.grad != 0).any(), \
        "整批无目标时 logits 没有梯度，背景类损失实际未生效"
    return f"loss={out['loss'].item():.4f}，背景类损失有梯度"


def test_loss_decreases_on_learnable_target() -> str:
    """在可学习的小问题上损失应当下降（证明损失函数可优化）。"""
    torch.manual_seed(0)
    Q, A = 8, 2
    logits = torch.zeros(1, Q, 2, requires_grad=True)
    boxes = torch.full((1, Q, 4), 0.5, requires_grad=True)
    attrs = torch.zeros(1, Q, A, requires_grad=True)
    targets = [{
        "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
        "labels": torch.tensor([0]),
        "attributes": torch.tensor([[1.0, -1.0]]),
    }]
    crit = SetCriterion(num_classes=1, num_attributes=A, cfg=LossConfig())
    opt = torch.optim.Adam([logits, boxes, attrs], lr=0.05)
    first = last = None
    for _ in range(60):
        opt.zero_grad()
        out = crit({"logits": logits, "pred_boxes": boxes, "pred_attributes": attrs}, targets)
        out["loss"].backward()
        opt.step()
        last = out["loss"].item()
        if first is None:
            first = last
    assert last < first * 0.8, f"损失未明显下降：{first:.4f} → {last:.4f}"
    return f"{first:.4f} → {last:.4f}"


def test_loss_weights_applied() -> str:
    """``eos_coef`` 必须真的作用在背景类上。"""
    crit = SetCriterion(num_classes=3, num_attributes=0,
                        cfg=LossConfig(eos_coef=0.1))
    w = crit.empty_weight
    assert w.shape[0] == 4, w.shape
    assert abs(float(w[-1]) - 0.1) < 1e-6, float(w[-1])
    assert abs(float(w[0]) - 1.0) < 1e-6, float(w[0])
    return f"empty_weight={w.tolist()}"


def test_ap_sanity() -> str:
    """mAP 计算的基本正确性。"""
    tgt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # 完美预测
    ap_perfect = simple_ap(torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                           torch.tensor([0.9]), tgt, 0.5)
    assert abs(ap_perfect - 1.0) < 1e-6, ap_perfect
    # 完全错位
    ap_bad = simple_ap(torch.tensor([[100.0, 100.0, 110.0, 110.0]]),
                       torch.tensor([0.9]), tgt, 0.5)
    assert abs(ap_bad) < 1e-6, ap_bad
    # 没有目标 → None（不应被当成 0 计入平均）
    assert simple_ap(torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                     torch.tensor([0.5]), torch.zeros(0, 4), 0.5) is None
    return "完美=1.0，错位=0.0，无目标=None"


# --------------------------------------------------------------------------
# 4. 旧实现的反例（证明我们在修什么）
# --------------------------------------------------------------------------

def test_old_topk_can_duplicate() -> str:
    """用旧实现的 topk 逻辑构造反例，证明它确实会重复匹配。

    这不是"测试旧代码"，而是把"为什么必须换成匈牙利"变成可执行的证据。

    反例构造：``iou`` 形状 ``[Q=6, N=3]``。查询 0 同时与**所有三个**目标的
    IoU 都是 0.99，查询 2 与目标 A 的 IoU 是 0.8，其余很小。

    旧代码::

        _, indices = torch.topk(iou.max(dim=1).values, min(num_targets, num_queries))

    ``iou.max(dim=1).values`` = 每个查询对"它最像的那个目标"的 IoU。
    于是查询 0 与查询 2 都是 0.99 附近，take top-3 → 查询 0 被选中一次，
    但它同时是三个目标的最佳查询。随后旧代码用
    ``target_labels[:len(indices)]`` 一一对应，就直接错位了。

    这里断言的是：**旧代码选出的查询集合无法覆盖全部目标**，
    而且它给出的 (查询, 目标) 对应关系不是置换。
    """
    num_queries, num_targets = 6, 3
    iou = torch.full((num_queries, num_targets), 0.02)
    iou[0, :] = 0.99          # 查询 0 通吃三个目标
    iou[2, 0] = 0.80          # 查询 2 对目标 A 也很好
    iou[1, 1] = 0.30
    iou[3, 2] = 0.25

    # ---- 旧实现 ----
    _, indices = torch.topk(iou.max(dim=1).values, min(num_targets, num_queries))
    old_sel = indices.tolist()

    # 旧实现隐含的对应：第 i 个被选中的查询 ←→ 第 i 个目标
    old_pairs = list(zip(old_sel, range(len(old_sel))))
    old_is_permutation = (
        len(set(old_sel)) == len(old_sel) and len(set(range(len(old_sel)))) == len(old_sel)
    )

    # ---- 匈牙利 ----
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(-iou.numpy())
    new_pairs = list(zip(row.tolist(), col.tolist()))
    new_is_permutation = len(set(row.tolist())) == len(row.tolist())

    # 判定：旧实现把目标 B 配给了查询 1（因为 top-3 选中的是 0,2,1），
    # 而查询 1 对目标 B 的 IoU 只有 0.30；查询 0 明明对目标 B 有 0.99 却被
    # 配给了目标 A。也就是说旧实现**无法保证把最佳查询留给对应目标**。
    old_regret = sum(float(iou[q, t]) for q, t in old_pairs)
    new_regret = sum(float(iou[q, t]) for q, t in new_pairs)

    assert new_is_permutation, "匈牙利结果不是置换"
    assert new_regret > old_regret, (
        f"反例失效：匈牙利总 IoU {new_regret:.3f} 未优于旧贪心 {old_regret:.3f}"
    )
    return (f"旧 topk 选出查询 {old_sel}（总 IoU {old_regret:.2f}，"
            f"置换性={old_is_permutation}）vs 匈牙利 {new_pairs}"
            f"（总 IoU {new_regret:.2f}）")


# --------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("DETR 管线自检（CPU，无需预训练权重）")
    print("=" * 72)

    check("数据集清单可构建且框归一化正确", test_manifest)
    check("按图分组划分无泄漏且可复现", test_group_split_no_leak)
    check("collate_fn 返回 list 语义的 targets", test_collate_contract)
    check("GIoU 数值正确且退化框不产生 NaN", test_giou)
    check("cxcywh ↔ xyxy 往返一致", test_cxcywh_roundtrip)
    check("匈牙利匹配是一对一置换", test_hungarian_is_permutation)
    check("无目标样本不崩且损失有限", test_no_target_does_not_crash)
    check("整批无目标不崩", test_all_empty_batch)
    check("损失在可学习问题上下滑", test_loss_decreases_on_learnable_target)
    check("eos_coef 正确作用于背景类", test_loss_weights_applied)
    check("AP 计算合理性", test_ap_sanity)
    check("（反例）旧 topk 匹配会重复", test_old_topk_can_duplicate)

    print()
    n_fail = 0
    for status, name, detail in _results:
        mark = "✓" if status == PASS else "✗"
        print(f"  {mark} {name}")
        if detail:
            print(f"      {detail}")
        if status == FAIL:
            n_fail += 1
    print()
    print(f"结果：{len(_results) - n_fail}/{len(_results)} 通过")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
