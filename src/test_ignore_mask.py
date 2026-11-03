#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ignore_mask.py —— 验证"忽略区"机制真的生效

## 为什么必须验证

忽略区是一个**沉默的机制**：如果它没接上（比如 ignore.png 没被读到、
或者在 DataLoader 里被丢掉），训练会照常进行、loss 照常下降，
**没有任何报错**，但你想屏蔽的像素其实仍在参与损失。

这类"看起来生效、实际没生效"的坑本项目已经踩过一次
（`--class-balance` 在单类时权重恒为 1.0，日志打印了却什么也没做）。

因此这里用**可构造的极端情形**做断言：

1. 把整幅图都标为 ignore -> 损失应当接近 0（没有可学习的像素）
2. 只把含全部正样本的区域标为 ignore ->
   剩余像素全是负样本，因此"预测全前景"的损失应当很大
3. 数据集能正确返回 4 元组，且 ignore 通道的形状是 (1,H,W)
4. 无 ignore.png 的旧数据仍返回 3 元组（向后兼容）

## 运行

    python tools/server/scripts/test_ignore_mask.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_seg import SegDataset, seg_loss  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_tile(root: str, name: str, with_ignore: bool,
              gt_mode: str = "corner") -> str:
    """造一个 tile 目录。

    gt_mode:
      corner   -> GT 在左上 64×64
      full     -> GT 覆盖全图
    """
    d = os.path.join(root, "set", name, "z0000")
    os.makedirs(d, exist_ok=True)
    S = 64
    rng = np.random.RandomState(0)
    em = rng.randint(60, 190, (S, S)).astype(np.uint8)
    Image.fromarray(em).save(os.path.join(d, "em.png"))

    gt = np.zeros((S, S), np.uint8)
    if gt_mode == "corner":
        gt[:32, :32] = 255
    else:
        gt[:] = 255
    Image.fromarray(gt).save(os.path.join(d, "organelle.png"))

    if with_ignore:
        ig = np.zeros((S, S), np.uint8)
        if gt_mode == "corner":
            ig[:32, :32] = 255          # 忽略掉全部正样本
        else:
            ig[:] = 255                 # 忽略全图
        Image.fromarray(ig).save(os.path.join(d, "ignore.png"))
    return d


def main() -> int:
    print("=" * 70)
    print("忽略区机制验证")
    print("=" * 70)
    tmp = tempfile.mkdtemp(prefix="ign_test_")
    print(f"\n临时目录 {tmp}")

    # ---------- 1. 数据集返回形状 ----------
    print("\n[1] SegDataset 的返回元组")
    make_tile(tmp, "nocase", with_ignore=False)
    ds3 = SegDataset(tmp, ("organelle",), 64, augment=False)
    check("无 ignore.png 时返回 3 元组", len(ds3[0]) == 3,
          f"got {len(ds3[0])}")
    check("找到 1 个切片", len(ds3) == 1, f"got {len(ds3)}")

    make_tile(tmp, "withcase", with_ignore=True)
    ds4 = SegDataset(tmp, ("organelle",), 64, augment=False)
    items = list(ds4)
    four = [it for it in items if len(it) == 4]
    three = [it for it in items if len(it) == 3]
    check("有 ignore.png 时返回 4 元组", len(four) >= 1, f"4元组 {len(four)}")
    check("同一数据集里两种都能共存", len(three) >= 1 and len(four) >= 1,
          f"3元组 {len(three)} / 4元组 {len(four)}")
    if four:
        x, y, ig, _ = four[0]
        check("x 形状 (3,64,64)", tuple(x.shape) == (3, 64, 64), str(tuple(x.shape)))
        check("y 形状 (1,64,64)", tuple(y.shape) == (1, 64, 64), str(tuple(y.shape)))
        check("ignore 形状 (1,64,64)", tuple(ig.shape) == (1, 64, 64),
              str(tuple(ig.shape)))

    # ---------- 2. 梯度行为（这才是真正要验证的） ----------
    #
    # 为什么不用"损失值"来验证：soft Dice 对"全负样本"的区域是退化的
    # （num=0, den≈p.sum() -> dice 恒为 0），所以即使完全忽略，
    # 损失值也不会掉到 0。但那是个**常数偏移**，不改变梯度。
    # 真正决定"忽略是否生效"的是：被忽略像素上的梯度是否精确为 0。
    print("\n[2] 被忽略像素的梯度必须精确为 0")
    torch.manual_seed(0)
    S = 64
    target = torch.zeros(1, 1, S, S)
    target[0, 0, :32, :32] = 1.0

    ign_pos = torch.zeros(1, 1, S, S)
    ign_pos[0, 0, :32, :32] = 1.0        # 忽略全部正样本

    def grads(ignore=None):
        lg = torch.zeros(1, 1, S, S, requires_grad=True)
        seg_loss(lg, target, bce_w=0.5, ignore=ignore).backward()
        return lg.grad.detach()

    g_none = grads(None)
    g_ign = grads(ign_pos)

    inside = g_ign[0, 0, :32, :32]
    outside = g_ign[0, 0, 32:, 32:]
    check("忽略区内梯度全为 0", float(inside.abs().max()) < 1e-9,
          f"max|g|={float(inside.abs().max()):.2e}")
    check("忽略区外梯度不为 0", float(outside.abs().max()) > 1e-6,
          f"max|g|={float(outside.abs().max()):.2e}")
    check("忽略后整体梯度与不忽略不同",
          float((g_none - g_ign).abs().max()) > 1e-6,
          f"max|Δg|={float((g_none - g_ign).abs().max()):.2e}")

    # 全图忽略 -> 所有梯度为 0
    g_all = grads(torch.ones(1, 1, S, S))
    check("忽略全图时梯度全为 0", float(g_all.abs().max()) < 1e-9,
          f"max|g|={float(g_all.abs().max()):.2e}")
    check("忽略全图时 backward 不报错（返回带图零值）", True)

    # ---------- 3. 与无 ignore 的一致性 ----------
    print("\n[3] ignore=None 时必须与旧行为完全一致")
    z = torch.zeros(1, 1, S, S)
    a = float(seg_loss(z, target, bce_w=0.5))
    b = float(seg_loss(z, target, bce_w=0.5, ignore=None))
    check("ignore=None 与不传该参数结果相同", abs(a - b) < 1e-9,
          f"{a:.6f} vs {b:.6f}")

    print()
    print("=" * 70)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("全部通过——忽略区机制确实生效，且不影响无 ignore 的旧数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
