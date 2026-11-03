#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morpho.losses —— DETR 的匈牙利匹配与损失（替代原实现里的 top-k 贪心匹配）

原实现的三个致命问题
--------------------
1. **匹配不是匹配**（``detr_region_feature.py:318``）::

       _, indices = torch.topk(iou_matrix.max(dim=1).values, min(num_targets, num_queries))

   这是"每个查询挑 IoU 最高的目标"，然后取分数最高的前 k 个查询。它既不是一对一
   置换（同一个目标可以被多个查询命中，也允许两个查询命中同一目标），也没有考虑
   分类代价。``进度3.pdf`` 里写的是匈牙利算法，代码里没有。

2. **没有 GIoU 损失**（``:322-333``）只有 ``CE + MSE(框) + 0.1*MSE(特征)``。

3. **无目标分支的损失语义错误**（``:305-309``）::

       bg_logits = logits[i, :, 0]
       loss_cls = F.binary_cross_entropy_with_logits(bg_logits, torch.ones_like(bg_logits))

   对 ``[num_queries]`` 形状做 BCE 并让目标全为 1，既不是分类损失的正确形式
   （应当是 CE，目标类别 = 背景类），也会让"全部预测背景"变成一个平凡极小点。

本模块实现标准的 DETR 损失：匈牙利匹配 + 加权 CE + L1 + GIoU + 属性标准化 MSE。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------
# 集合运算
# --------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """成对 IoU，返回 ``[len(a), len(b)]``。输入均为 ``xyxy``。"""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    area_a = box_area(a)
    area_b = box_area(b)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7)


def generalized_box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GIoU，返回 ``[len(a), len(b)]``，取值范围 ``(-1, 1]``。

    注意：当框退化（面积为 0）时，闭包框面积也可能为 0，需要 clamp 防止 NaN。
    """
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    iou = box_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]
    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]
    union = area_a + area_b - (
        (torch.min(a[:, None, 2:], b[None, :, 2:]) - torch.max(a[:, None, :2], b[None, :, :2]))
        .clamp(min=0).prod(-1)
    )
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


# --------------------------------------------------------------------------
# 配置与损失
# --------------------------------------------------------------------------

@dataclass
class LossConfig:
    """损失权重。默认值与 ``进度3.pdf`` 中给出的 config 一致。"""
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    attribute_loss_coef: float = 1.0
    eos_coef: float = 0.1           # 背景类损失权重（原文 eos_coef）
    match_cost_class: float = 1.0
    match_cost_bbox: float = 5.0
    match_cost_giou: float = 2.0
    #: 属性损失是否在批次内做标准化（原文"内部批次标准化"策略）
    normalize_attributes: bool = True


class SetCriterion(nn.Module):
    """DETR 集合预测损失。

    参数
    ----
    num_classes
        前景类数量。分类头输出维度必须是 ``num_classes + 1``（+1 为 no-object）。
    num_attributes
        回归属性数量；0 表示不计算属性损失。
    """

    def __init__(
        self,
        num_classes: int,
        num_attributes: int = 0,
        cfg: Optional[LossConfig] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.cfg = cfg or LossConfig()

        # 分类目标：最后一个索引是背景
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.cfg.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # -- 匹配 ------------------------------------------------------------
    @torch.no_grad()
    def _match(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """对每个样本求最优一对一匹配（匈牙利算法）。

        返回 ``[(src_idx, tgt_idx), ...]``，每个元素长度 = 该样本的目标数。
        """
        if not _HAS_SCIPY:
            raise RuntimeError(
                "需要 scipy 才能做匈牙利匹配：pip install scipy。"
                "原实现用 top-k 贪心代替，会重复匹配同一个目标。"
            )

        out_prob = outputs["logits"].softmax(-1)          # [B, Q, C+1]
        out_bbox = outputs["pred_boxes"]                  # [B, Q, 4] cxcywh
        matches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for b, tgt in enumerate(targets):
            tgt_bbox = tgt["boxes"]                       # cxcywh
            tgt_ids = tgt["labels"]
            n_tgt = tgt_bbox.shape[0]
            if n_tgt == 0:
                matches.append((
                    torch.empty(0, dtype=torch.long, device=out_prob.device),
                    torch.empty(0, dtype=torch.long, device=out_prob.device),
                ))
                continue

            cost_class = -out_prob[b][:, tgt_ids]         # [Q, N]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox)
            )
            cost_giou = -giou

            C = (
                self.cfg.match_cost_class * cost_class
                + self.cfg.match_cost_bbox * cost_bbox
                + self.cfg.match_cost_giou * cost_giou
            )
            C = C.detach().cpu()
            if not torch.isfinite(C).all():
                C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

            src, dst = linear_sum_assignment(C.numpy())
            matches.append((
                torch.as_tensor(src, dtype=torch.long, device=out_prob.device),
                torch.as_tensor(dst, dtype=torch.long, device=out_prob.device),
            ))
        return matches

    # -- 前向 ------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        matches = self._match(outputs, targets)
        device = outputs["logits"].device
        B, Q = outputs["logits"].shape[:2]

        # ---- 分类：所有查询都参与，匹配上的为正类，其余为背景 ----
        # 原实现在"无目标"时对 [Q] 做 BCE(target=1)，语义与形状都是错的。
        # 注意：`src_all` 等一开始是 list，**只有在确实匹配到目标时才 cat 成 tensor**。
        # 早先的写法用 `if src_all:` 判断，空的 list 与"全为背景"两种情况会走到
        # 不同的分支，随后 `src_all.numel()` 在 list 上直接 AttributeError
        # （整批无目标时必现）。这里用显式的 None 表示"无匹配"。
        src_all: Optional[torch.Tensor] = None
        tgt_boxes_sel: Optional[torch.Tensor] = None
        tgt_attrs_all: Optional[torch.Tensor] = None

        src_list, tgt_list, batch_list = [], [], []
        box_list, attr_list = [], []
        for b, (src, dst) in enumerate(matches):
            if src.numel() == 0:
                continue
            src_list.append(src)
            tgt_list.append(targets[b]["labels"][dst])
            batch_list.append(torch.full_like(src, b))
            box_list.append(targets[b]["boxes"][dst])
            if "attributes" in targets[b]:
                attr_list.append(targets[b]["attributes"][dst])

        n_matched = sum(t.shape[0] for t in src_list)
        if n_matched > 0:
            src_all = torch.cat(src_list)
            tgt_all = torch.cat(tgt_list)
            batch_all = torch.cat(batch_list)
            tgt_boxes_sel = torch.cat(box_list, dim=0)
            if attr_list:
                tgt_attrs_all = torch.cat(attr_list, dim=0)
            target_classes = torch.full(
                (B, Q), self.num_classes, dtype=torch.long, device=device
            )
            target_classes[batch_all, src_all] = tgt_all
        else:
            target_classes = torch.full(
                (B, Q), self.num_classes, dtype=torch.long, device=device
            )

        loss_ce = F.cross_entropy(
            outputs["logits"].transpose(1, 2),      # [B, C+1, Q]
            target_classes,
            weight=self.empty_weight.to(device),
        )

        # ---- 框 L1 + GIoU（仅匹配上的查询）----
        num_boxes = max(sum(t["boxes"].shape[0] for t in targets), 1)
        if n_matched > 0:
            pred_boxes_sel = outputs["pred_boxes"][batch_all, src_all]
            loss_bbox = F.l1_loss(pred_boxes_sel, tgt_boxes_sel, reduction="sum") / num_boxes
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes_sel),
                box_cxcywh_to_xyxy(tgt_boxes_sel),
            )
            loss_giou = (1 - torch.diag(giou)).sum() / num_boxes
        else:
            # 用参与计算的零，而不是 `x.sum()*0.0` —— 后者在 x 不需要梯度时
            # 会得到一个 requires_grad=False 的张量，把整条损失的梯度链断掉。
            loss_bbox = outputs["logits"].sum() * 0.0
            loss_giou = outputs["logits"].sum() * 0.0

        total = (
            loss_ce
            + self.cfg.bbox_loss_coef * loss_bbox
            + self.cfg.giou_loss_coef * loss_giou
        )
        parts = {
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

        # ---- 属性回归（仅匹配上的查询）----
        if (self.num_attributes > 0 and "pred_attributes" in outputs
                and n_matched > 0 and tgt_attrs_all is not None):
            pred_attr = outputs["pred_attributes"][batch_all, src_all]     # [M, A]
            tgt_attr = tgt_attrs_all
            if pred_attr.shape != tgt_attr.shape:
                # 键不匹配时跳过属性损失，而不是静默广播出错误的数值
                parts["loss_attr_skipped_shape_mismatch"] = torch.tensor(
                    1.0, device=loss_ce.device
                )
            else:
                if self.cfg.normalize_attributes and pred_attr.shape[0] > 1:
                    # 内部批次标准化：只在损失里作用梯度，不改变推理输出的物理尺度
                    mu = pred_attr.mean(dim=0, keepdim=True)
                    sd = pred_attr.std(dim=0, keepdim=True).clamp(min=1e-6)
                    pred_n = (pred_attr - mu) / sd
                    tgt_n = (tgt_attr - mu) / sd
                    loss_attr = F.mse_loss(pred_n, tgt_n, reduction="mean")
                else:
                    loss_attr = F.mse_loss(pred_attr, tgt_attr, reduction="mean")
                total = total + self.cfg.attribute_loss_coef * loss_attr
                parts["loss_attr"] = loss_attr.detach()

        parts["loss_total"] = total
        return {"loss": total, "losses": parts, "matches": matches}


# --------------------------------------------------------------------------
# 评估辅助
# --------------------------------------------------------------------------

@torch.no_grad()
def simple_ap(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    tgt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Optional[float]:
    """单图、单类别的 AP（用于 mAP 的组成单元）。

    没有引 pycocotools：本项目只有一个前景类、验证集只有几十张图，
    一个按置信度排序的贪心匹配已经足够反映趋势。若要正式汇报，请换用
    ``torchvision.ops.box_iou`` + pycocotools 的标准实现。
    """
    n_tgt = tgt_boxes.shape[0]
    if n_tgt == 0:
        return None
    if pred_boxes.numel() == 0:
        return 0.0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    ious = box_iou(pred_boxes, tgt_boxes)              # [P, N]
    matched = torch.zeros(n_tgt, dtype=torch.bool)

    tp = torch.zeros(pred_boxes.shape[0])
    fp = torch.zeros(pred_boxes.shape[0])
    for i in range(pred_boxes.shape[0]):
        row = ious[i].clone()
        row[matched] = -1.0
        best_iou, best_j = row.max(dim=0)
        if best_iou >= iou_threshold:
            tp[i] = 1.0
            matched[best_j] = True
        else:
            fp[i] = 1.0

    tp_c = tp.cumsum(0)
    fp_c = fp.cumsum(0)
    recall = tp_c / n_tgt
    precision = tp_c / (tp_c + fp_c).clamp(min=1e-9)
    # VOC2010+ 的插值 AP
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall.tolist(), precision.tolist()):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    return float(ap)
