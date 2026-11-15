#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resnet_unet.py —— 以预训练 ResNet-50 为编码器的 U-Net

## 为什么做这个

本项目此前所有分割模型都是从零训练的 2.29 M 参数小 U-Net。四条改进路线全部为负，
共同的机制是「公开数据（OpenOrganelle）的特征与自有 TEM 插图不可通约」：
公开数据是单一 FIB-SEM 模态，编码器学到的判别式在目标域上失效，
且负效应随对该权重的锚定强度单调加深。

CEM500K 换了一个前提：它在 **50 万张异构细胞电镜图**上用 MoCo v2 自监督训练，
目的就是学「通用的电镜纹理」而非某一模态的判别式。若瓶颈确实是「缺通用电镜特征」，
那么以它为编码器应当优于从零训练——这是一个可以判定真假的假设。

## 加载 MoCo 权重

MoCo 的 checkpoint 通常把编码器存在 `state_dict` 里，键名带 `module.encoder_q.` 前缀。
不同发布版本的前缀不一致（`encoder_q.`、`module.`、`backbone.` 都有），
所以这里用**逐键剥离**的方式：反复去掉已知前缀，直到键能对上为止，并报告匹配率。
匹配率过低时显式告警——静默只载入 3% 的权重会让结论完全错误。
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn

KNOWN_PREFIXES = (
    "module.encoder_q.", "encoder_q.", "module.encoder.", "encoder.",
    "module.backbone.", "backbone.", "module.", "model.",
)


def _strip(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in KNOWN_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def load_moco_encoder(path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """从 MoCo / 普通 checkpoint 里取出可以按 ResNet 键名匹配的权重。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("state_dict", "model", "model_state", "encoder"):
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
    if not isinstance(ck, dict):
        raise SystemExit(f"[错误] {path} 里读不到状态字典")

    out: Dict[str, torch.Tensor] = {}
    for k, v in ck.items():
        if not torch.is_tensor(v):
            continue
        k2 = _strip(k)
        # 只要 ResNet 主干与 stem 的键；head/fc 用不到
        if k2.startswith(("conv1", "bn1", "layer", "maxpool")):
            out[k2] = v
    return out, {"total_keys": len(ck), "kept": len(out)}


def double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class ResNet50UNet(nn.Module):
    """ResNet-50 编码器 + 四级 U-Net 解码器。

    编码器输出四个尺度（stride 4 / 8 / 16 / 32），解码器逐级上采样并拼接。
    输入 256×256 时最深层为 8×8。
    """

    def __init__(self, cout: int = 1, pretrained: str | None = None):
        super().__init__()
        import torchvision
        # pretrained="imagenet" 用于对照：外部预训练到底有没有用，
        # 还是只有"电镜域的预训练"没用。ImageNet 与电镜差得最远，
        # 若它反而有帮助，说明此前的失败是域特定问题而非预训练本身。
        if pretrained == "imagenet":
            r = torchvision.models.resnet50(weights="IMAGENET1K_V1")
            self.load_report = {"path": "imagenet", "matched": -1,
                                "unmatched": -1, "match_rate": 1.0}
            self._skip_load = True
        else:
            r = torchvision.models.resnet50(weights=None)
            self._skip_load = False

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2,  64
        self.pool = r.maxpool                                  # /4
        self.layer1 = r.layer1                                 # /4,  256
        self.layer2 = r.layer2                                 # /8,  512
        self.layer3 = r.layer3                                 # /16, 1024
        self.layer4 = r.layer4                                 # /32, 2048

        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.c4 = double_conv(1024 + 1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.c3 = double_conv(512 + 512, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.c2 = double_conv(256 + 256, 128)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.c1 = double_conv(128 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c0 = double_conv(32, 32)
        self.out = nn.Conv2d(32, cout, 1)

        if not getattr(self, "_skip_load", True) and pretrained:
            sd, info = load_moco_encoder(pretrained)
            own = self.state_dict()
            hit, miss = 0, []
            for k, v in sd.items():
                if k in own and tuple(own[k].shape) == tuple(v.shape):
                    own[k] = v
                    hit += 1
                else:
                    miss.append(k)
            self.load_state_dict(own)
            total = hit + len(miss)
            self.load_report = {
                "path": pretrained,
                "matched": hit,
                "unmatched": len(miss),
                "match_rate": (hit / total) if total else 0.0,
                **info,
            }
            rate = self.load_report["match_rate"]
            print(f"  CEM500K 编码器：匹配 {hit}/{total} 个张量"
                  f"（{100*rate:.1f}%），来源 {pretrained}")
            if rate < 0.8:
                print(f"  [告警] 匹配率仅 {100*rate:.1f}%，低于 80%。"
                      f"编码器大部分仍是随机初始化，结论不可解释。")
                print(f"          前几个未匹配键：{miss[:4]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)                 # /2,  64
        p = self.pool(s)                 # /4,  64
        f1 = self.layer1(p)              # /4,  256
        f2 = self.layer2(f1)             # /8,  512
        f3 = self.layer3(f2)             # /16, 1024
        f4 = self.layer4(f3)             # /32, 2048
        u = self.c4(torch.cat([self.up4(f4), f3], 1))
        u = self.c3(torch.cat([self.up3(u), f2], 1))
        u = self.c2(torch.cat([self.up2(u), f1], 1))
        u = self.c1(torch.cat([self.up1(u), s], 1))
        u = self.c0(self.up0(u))
        return self.out(u)


if __name__ == "__main__":
    # 自检：能在给定权重下正确加载并前向
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    m = ResNet50UNet(1, a.weights)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"参数量 {n:.1f} M")
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 256, 256))
    print("前向输出形状:", tuple(y.shape))
    print("加载报告:", m.load_report)
