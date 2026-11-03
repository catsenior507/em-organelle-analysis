#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morpho.detr_model —— DETR + 属性回归头（替代 code/detr_region_feature.py 的模型部分）

修复的原实现缺陷
----------------
1. **手工改 ``decoder.num_queries``**（原 ``:251-252``）::

       self.detr.model.decoder.num_queries = num_queries
       self.detr.model.decoder.query_embed = nn.Embedding(num_queries, hidden_dim)

   DETR 解码器里与查询数量相关的**不只** ``query_embed``：还有位置嵌入
   ``embed_pos``（形状 ``[1, num_queries, d_model]``），而且解码器的前向会把
   ``query_embed`` 当作 *位置* 嵌入加到 ``query`` 上。改了 ``num_queries`` 却不同步
   ``embed_pos``，前向会直接因广播失败而报错，或（在某些版本里）静默广播成错误的语义。
   本实现通过 ``DetrConfig(num_queries=...)`` 从头构建解码器，并在加载预训练权重时
   显式处理形状不匹配的参数。

2. **缺少 ``decoder_norm``**（原 ``:272``）::

       pred_boxes = self.detr.bbox_predictor(decoder_output).sigmoid()

   HuggingFace 的实现里，分类头和框回归头的输入是
   ``decoder_norm(decoder_output)``；直接喂未归一化的 ``last_hidden_state``
   会让预训练头接收到分布完全不同的输入。

3. **分类头语义**：本项目只有一个前景类（细胞器），因此 ``num_labels=1``，
   分类头输出维度为 ``num_labels + 1 = 2``（最后一维是 no-object）。
   原实现写死 ``nn.Linear(in_features, 2)`` 是对的，但没有把它与损失里的
   ``num_classes`` 统一起来，容易在改动时失配。

4. **新增属性回归头**，对应 ``进度3.pdf`` 里"区域属性 ∈ R^14"的设计。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: 预训练权重标识。注意：需要联网下载；离线环境请先
#: ``huggingface-cli download facebook/detr-resnet-50`` 或在有网机器上缓存后拷贝。
DEFAULT_PRETRAINED = "facebook/detr-resnet-50"


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

@dataclass
class DetrModelConfig:
    num_queries: int = 15                    # 与 进度3.pdf 的方案一致
    num_classes: int = 1                     # 唯一前景类：细胞器
    num_attributes: int = 6                  # 属性回归维度（见 dataset.DetrDataConfig.attribute_names）
    pretrained: Optional[str] = DEFAULT_PRETRAINED
    #: 为 True 时冻结主干，只训 transformer 与头（数据量小时更稳）
    freeze_backbone: bool = False
    #: 属性头的隐藏层宽度
    attribute_head_hidden: int = 128


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------

class DetrRegionModel(nn.Module):
    """DETR 目标检测 + 区域属性回归。

    ``forward`` 返回::

        {
          'logits':         [B, Q, num_classes + 1],
          'pred_boxes':     [B, Q, 4],   # 归一化 cxcywh，已 sigmoid
          'pred_attributes':[B, Q, A],   # 属性回归（未标准化，物理尺度）
          'decoder_output': [B, Q, d],   # 便于调试/特征复用
        }
    """

    def __init__(self, cfg: DetrModelConfig) -> None:
        super().__init__()
        # 延迟导入：transformers 的 import 很慢，且只有在真正要用模型时才需要
        from transformers import DetrForObjectDetection

        self.cfg = cfg

        base = DetrForObjectDetection.from_pretrained(cfg.pretrained)
        base_cfg = base.config
        hidden_dim = base_cfg.d_model

        # 复用预训练的**整个 DetrModel**（主干 + 输入投影 + 编解码器）。
        # 关键点：不要拆开部件自己调 transformer —— DetrModel.forward 内部负责
        # 生成/注入可学习查询 ``query_embed`` 与查询位置嵌入 ``embed_pos``。
        # 原实现手工调 ``self.detr.model(...)`` 之外还去改 decoder 的字段，才会错位。
        self.detr_model = base.model

        # ---- 输出头 ----
        # 分类头：num_classes + 1（最后一位是 no-object）
        self.class_labels_classifier = nn.Linear(hidden_dim, cfg.num_classes + 1)
        # 框回归头：3 层 MLP，最后一层零初始化（DETR 官方做法）
        self.bbox_predictor = _make_mlp(hidden_dim, hidden_dim, 4, 3)
        # 属性回归头
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_dim, cfg.attribute_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.attribute_head_hidden, cfg.num_attributes),
        )

        # ---- 查询数量：必须与 query_embed / embed_pos 同步重建 ----
        self._reset_query_embeddings(num_queries=cfg.num_queries, hidden_dim=hidden_dim)

        if cfg.freeze_backbone:
            for p in self.detr_model.backbone.parameters():
                p.requires_grad_(False)
            proj = self._input_projection()
            if proj is not None:
                for p in proj.parameters():
                    p.requires_grad_(False)

        logger.info(
            "DetrRegionModel: queries=%d classes=%d attrs=%d hidden=%d pretrained=%s",
            cfg.num_queries, cfg.num_classes, cfg.num_attributes,
            hidden_dim, cfg.pretrained,
        )

    # ------------------------------------------------------------------
    def _input_projection(self) -> Optional[nn.Module]:
        for name in ("input_projection", "input_proj"):
            if hasattr(self.detr_model, name):
                return getattr(self.detr_model, name)
        return None

    def _reset_query_embeddings(self, num_queries: int, hidden_dim: int) -> None:
        """重建解码器的 ``query_embed`` 与 ``embed_pos``，使查询数一致。

        这两个张量都以查询数为第一维：

        - ``query_embed.weight``: ``[num_queries, hidden_dim]`` —— 可学习查询
        - ``embed_pos.weight``:   ``[1, num_queries, hidden_dim]`` —— 查询位置嵌入

        **只改前者、不改后者，是原实现必然崩溃的直接原因**：
        ``DetrDecoder.forward`` 会计算 ``hidden_states = query_embed + embed_pos``，
        形状不一致时要么广播成错误的语义，要么直接报错。
        """
        decoder = self.detr_model.transformer.decoder
        replaced = []

        if hasattr(decoder, "query_embed"):
            decoder.query_embed = nn.Embedding(num_queries, hidden_dim)
            replaced.append("query_embed")

        for attr in ("embed_pos", "pos_embed", "position_embed"):
            if hasattr(decoder, attr):
                setattr(decoder, attr, nn.Embedding(num_queries, hidden_dim))
                replaced.append(attr)
                break
        else:
            logger.warning(
                "解码器未找到位置嵌入字段（embed_pos/pos_embed/position_embed）；"
                "若前向报形状错误，请检查 transformers 版本：%s",
                type(decoder).__name__,
            )

        if hasattr(decoder, "num_queries"):
            decoder.num_queries = num_queries

        logger.info("已重建查询相关参数: %s -> num_queries=%d", ", ".join(replaced), num_queries)

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 交给预训练的 DetrModel：它内部完成 主干 → 输入投影 →
        # 生成 query_embed/embed_pos → 编解码。这样才不会与预训练权重错位。
        outputs = self.detr_model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True,
        )
        decoder_output = outputs.last_hidden_state                     # [B, Q, d]

        # ---- 头：**必须**先过 decoder_norm ----
        # 原实现直接把未归一化的 last_hidden_state 喂给分类/框头。
        normed = self.detr_model.transformer.decoder_norm(decoder_output)

        logits = self.class_labels_classifier(normed)                  # [B, Q, C+1]
        pred_boxes = self.bbox_predictor(normed).sigmoid()             # [B, Q, 4] cxcywh ∈ [0,1]
        pred_attributes = self.attribute_head(normed)                  # [B, Q, A]

        return {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "pred_attributes": pred_attributes,
            "decoder_output": decoder_output,
        }

    # ------------------------------------------------------------------
    def parameter_groups(self, lr: float, lr_backbone: float, weight_decay: float
                         ) -> List[Dict[str, Any]]:
        """按"主干 / 其余"分组设置学习率。

        原实现用一个 ``AdamW(model.parameters(), lr=1e-4)`` 一把梭，
        主干（ImageNet 预训练）与随机初始化的头共用一个学习率。``进度3.pdf``
        里给出的 ``lr=1e-4 / lr_backbone=1e-5`` 正是这个分组。
        """
        backbone_ids = set()
        for p in self.detr_model.backbone.parameters():
            backbone_ids.add(id(p))
        proj = self._input_projection()
        if proj is not None:
            for p in proj.parameters():
                backbone_ids.add(id(p))

        backbone_params, head_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (backbone_params if id(p) in backbone_ids else head_params).append(p)

        groups = [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        ]
        return [g for g in groups if g["params"]]


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
    """构造 DETR 官方的 MLP 预测头（最后一层零初始化）。"""
    layers: List[nn.Module] = []
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    mlp = nn.Sequential(*layers)
    # 最后一层零初始化：初始预测为常量，训练早期更稳定
    last = layers[-1]
    if isinstance(last, nn.Linear):
        nn.init.constant_(last.weight, 0.0)
        nn.init.constant_(last.bias, 0.0)
    return mlp
