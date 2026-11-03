#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py —— 端到端流水线：从原始图像到网络分析结果

## 这个脚本解决的问题

在此之前，各环节是**十多个独立脚本**，每个有自己的参数、自己的输出路径、
自己的前置条件。要跑通一遍得记住调用顺序和参数，换个人（或换台机器）
几乎无法复现。

本脚本把它们组织成**一条有序、可重复、可断点续跑**的流水线：
每一步都声明自己的输入、输出与是否已完成，`--from` / `--only` 可单独重跑。

## 步骤

    0  env        检查环境与依赖（Python 版本、torch、可选 empanada）
    1  tiles      原始图像 + 绿色标注  ->  256×256 tile 数据集
    2  train      训练分割模型（自有 U-Net，或微调 MitoNet）
    3  predict    全图滑窗推理 -> 原分辨率二值掩码
    4  filter     按物理面积过滤碎片
    5  regions    掩码 -> annotations_v2 同构的 regions JSON
    6  network    网络分析（必须用 radius 判据，见 docs）
    7  report     汇总所有指标，生成一份可读报告

## 典型用法

    # 全流程（已有产物会自动跳过）
    python tools/run_pipeline.py --steps all

    # 只重跑推理之后的步骤（换了模型时）
    python tools/run_pipeline.py --from predict

    # 只跑某几步
    python tools/run_pipeline.py --only filter,regions

    # 看每步状态，不执行
    python tools/run_pipeline.py --status

    # 强制重跑
    python tools/run_pipeline.py --from predict --force

## 设计约定

* **幂等**：每一步执行前检查输出是否已存在，存在则跳过（除非 --force）
* **可观测**：每步打印输入/输出/耗时，并写入 runs/pipeline_manifest.json
* **不重复实现**：每步都调用既有工具脚本，避免逻辑分叉
* **失败了就停**：某步非零退出立即中止，不带着半成品往下走
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "runs", "pipeline_manifest.json")

# 线粒体的默认权重（发布版）。高尔基体在 main 里会按数据集改写成自己的权重，
# 否则 `--dataset ourdata_golgi` 会静默地用线粒体模型去推理高尔基体图——
# 一个不会报错但完全错误的组合。
DEFAULT_CKPT = "models/ourdata_release/best.pt"

# 解释器：优先用项目 .venv（empanada 与 torch 都装在那里），否则用当前
_VENV = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if os.path.exists(_VENV):
    PY = _VENV
else:
    PY = sys.executable


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def sh(args: List[str], desc: str = "") -> int:
    """执行子命令并实时回显。返回退出码。"""
    if desc:
        print(f"    $ {' '.join(args)}")
    p = subprocess.run(args, cwd=ROOT)
    return p.returncode


def exists(*paths: str) -> bool:
    return all(os.path.exists(os.path.join(ROOT, p)) for p in paths)


def count(pattern: str) -> int:
    import glob
    return len(glob.glob(os.path.join(ROOT, pattern)))


def load_manifest() -> Dict:
    if os.path.exists(MANIFEST):
        try:
            return json.load(open(MANIFEST, encoding="utf-8"))
        except Exception:
            pass
    return {"steps": {}}


def save_manifest(m: Dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------
# 各步骤：完成判定 + 执行
# --------------------------------------------------------------------------
def step_env(args) -> int:
    """检查环境。不产生产物，永远执行。"""
    print(f"    Python: {PY}")
    rc = sh([PY, "-c", "import sys,torch;print('    torch',torch.__version__,"
                       "'cuda',torch.cuda.is_available())"])
    if rc != 0:
        print("    [失败] 无法导入 torch")
        return rc
    # empanada 可选
    sh([PY, "-c",
        "try:\n"
        "    import empanada; print('    empanada 可用')\n"
        "except Exception:\n"
        "    print('    empanada 未安装（MitoNet 相关步骤不可用）')"])
    return 0


def step_tiles(args) -> int:
    """建 tile 数据集。走统一层 `tools/unified.py`，按 `datasets.yaml` 的声明处理。

    早期这里直接调 `build_ourdata_tiles.py`（只认调色板 TIFF + 独立绿掩码）。
    现在线粒体和高尔基体两类数据、两种标注形态都从同一入口走，
    参数来自声明文件，不再按数据集分支。
    """
    return sh([PY, "tools/unified.py", "build", args.dataset,
               "--out", args.tiles_out])


def step_train(args) -> int:
    """训练。--trainer 选择：unet（从零）或 mitonet（微调）。"""
    if args.trainer == "mitonet":
        print("    MitoNet 微调需要在 GPU 服务器上运行：")
        print("      python tools/server/push_run.py tools/server/scripts/launch_finetune.sh")
        print("    本机只做数据准备：")
        return sh([PY, "tools/build_empanada_dataset.py"])
    return sh([PY, "src/train_seg.py",
               "--train-root", os.path.join(args.tiles_out, args.dataset + "_tiles"),
               "--val-root", "none",
               "--classes", args.cls, "--size", "256", "--batch", "16",
               "--epochs", "50", "--lr", "3e-4", "--base", "32",
               "--workers", "4", "--seed", "0",
               "--out", "runs/seg_ourdata_release", "--resume", "none"])


def step_predict(args) -> int:
    """全图推理。按数据集选推理脚本。

    线粒体与高尔基体的推理**不能共用一份代码**：
    * 线粒体那批有可靠的 nm/px（`annotations_v2/dataset.json`），
      推理时会统一重采样到 4 nm；
    * 高尔基体插图没有尺度标定，必须在**原分辨率**上推理，
      否则等于凭空假设一个尺度。
    此外两者的输出字段与阈值也不同。

    早期这里硬编码 `predict_all_ourdata.py`，于是 `--dataset ourdata_golgi`
    也能"跑通"，但用的是线粒体的重采样逻辑——一个不会报错的错误。
    """
    is_golgi = "golgi" in args.dataset.lower()
    if is_golgi:
        cmd = [PY, "tools/predict_golgi.py",
               "--ckpt", args.ckpt,
               "--out", args.masks_raw,
               "--orig", args.golgi_orig,
               "--annot", args.golgi_annot]
        if args.overlays:
            cmd += ["--overlays", args.overlays]
        return sh(cmd)

    cmd = [PY, "tools/predict_all_ourdata.py",
           "--ckpt", args.ckpt,
           "--out", args.masks_raw]
    # 验证名单只有线粒体那批固定过；高尔基体按 spec 的 val_every 划分，
    # 这里不传名单（predict 会把全部图当作 train，仅影响 index.csv 的标签列）
    vl = args.val_list
    if vl and os.path.exists(os.path.join(ROOT, vl)):
        cmd += ["--val-list", vl]
    return sh(cmd)


def step_filter(args) -> int:
    return sh([PY, "tools/filter_masks.py",
               "--src", args.masks_raw,
               "--dst", args.masks_clean,
               "--min-area-um2", str(args.min_area_um2)])


def step_regions(args) -> int:
    return sh([PY, "tools/seg_to_regions.py",
               "--masks", args.masks_clean,
               "--organelle", args.cls,
               "--out", args.annotations_out])


def step_network(args) -> int:
    # 必须用 radius：预测实例比标注小约 10%，硬接触判据会把边数打到 1/35
    return sh([PY, "tools/network/run_analysis.py",
               "--annotations", args.annotations_out,
               "--out", args.network_out,
               "--method", "radius"])


def step_report(args) -> int:
    return sh([PY, "tools/pipeline_report.py",
               "--dataset", args.dataset,
               "--annotations", args.annotations_out,
               "--network", args.network_out])


# 步骤定义：名称 -> (说明, 函数, 完成判定所需的产物, 是否允许跳过)
# 步骤的"产物"按参数动态计算，因此两类数据可以并存而互不覆盖。
# 早期版本把路径写死在表里，导致 --dataset 换了之后状态显示仍是线粒体的。
def _outputs(name: str, args) -> List[str]:
    if name == "tiles":
        return [os.path.join(args.tiles_out, args.dataset + "_tiles", "manifest.json")]
    if name == "train":
        return [args.ckpt]
    if name == "predict":
        return [os.path.join(args.masks_raw, "index.csv")]
    if name == "filter":
        return [os.path.join(args.masks_clean, "index.csv")]
    if name == "regions":
        return [os.path.join(args.annotations_out, "dataset.json")]
    if name == "network":
        return [os.path.join(args.network_out, "summary.json")]
    if name == "report":
        return ["logs/PIPELINE_REPORT.md"]
    return []


STEPS: List[tuple] = [
    ("env",     "环境检查",                  step_env,     False),
    ("tiles",   "建 tile 数据集（统一层）",   step_tiles,   True),
    ("train",   "训练/微调模型",             step_train,   True),
    ("predict", "全图推理",                  step_predict, True),
    ("filter",  "按面积过滤碎片",            step_filter,  True),
    ("regions", "掩码 -> regions JSON",      step_regions, True),
    ("network", "网络分析",                  step_network, True),
    ("report",  "汇总报告",                  step_report,  True),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="端到端流水线：原始图像 -> 掩码 -> 形态学 -> 网络分析")
    ap.add_argument("--steps", default=None,
                    help="逗号分隔的步骤名，或 all")
    ap.add_argument("--only", default=None, help="只跑这些步骤（同 --steps）")
    ap.add_argument("--from", dest="from_step", default=None,
                    help="从该步骤开始跑到最后")
    ap.add_argument("--force", action="store_true", help="忽略已有产物，强制重跑")
    ap.add_argument("--status", action="store_true", help="只报告每步状态")
    ap.add_argument("--dataset", default="ourdata_mito",
                    help="datasets.yaml 里的数据集名，决定输入形态与类别")
    ap.add_argument("--cls", default=None,
                    help="细胞器类别（mito/golgi）；缺省从数据集名推断")
    ap.add_argument("--tiles-out", default="data",
                    help="tile 输出根目录；实际路径为 <root>/<dataset>_tiles")
    ap.add_argument("--annotations-out", default=None,
                    help="regions JSON 输出目录；缺省按数据集派生")
    ap.add_argument("--network-out", default=None,
                    help="网络分析输出目录；缺省按数据集派生")
    ap.add_argument("--val-list", default="models/ourdata_mito/val_images.txt",
                    help="验证名单（仅在文件存在时传给 predict）")
    ap.add_argument("--golgi-orig", default="data/golgi/原图",
                    help="高尔基体原图目录（predict 用）")
    ap.add_argument("--golgi-annot", default="data/golgi/标注",
                    help="高尔基体标注目录（predict 用于计算 Dice）")
    ap.add_argument("--overlays", default="",
                    help="叠加图输出目录（仅高尔基体推理使用）")
    ap.add_argument("--trainer", default="unet", choices=["unet", "mitonet"])
    # 默认权重常量：高尔基体会在 main 里按数据集改写成自己的权重
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--masks-raw", default="data/masks_ourdata")
    ap.add_argument("--masks-clean", default="data/masks_ourdata_clean")
    ap.add_argument("--min-area-um2", type=float, default=0.05)
    args = ap.parse_args(argv)

    # 类别缺省从数据集名推断（ourdata_mito -> mito）
    if not args.cls:
        args.cls = args.dataset.rsplit("_", 1)[-1] if "_" in args.dataset else "mito"
    # 各步骤的产物路径随数据集走，避免两类数据互相覆盖
    if args.dataset != "ourdata_mito":
        suffix = args.dataset
        args.masks_raw = f"data/masks_{suffix}"
        args.masks_clean = f"data/masks_{suffix}_clean"
        if not args.annotations_out:
            args.annotations_out = f"data/annotations_{suffix}"
        if not args.network_out:
            args.network_out = f"data/network_{suffix}"
    # 高尔基体有自己的权重目录；--ckpt 未显式指定时按数据集派生
    if args.ckpt == DEFAULT_CKPT and "golgi" in args.dataset.lower():
        for cand in ("runs/golgi_v3/best.pt", "runs/seg_golgi_combined/best.pt",
                     "runs/seg_golgi_own/best.pt"):
            if os.path.exists(os.path.join(ROOT, cand)):
                args.ckpt = cand
                break
    if not args.annotations_out:
        args.annotations_out = "data/annotations_seg"
    if not args.network_out:
        args.network_out = "data/network_analysis_seg_radius"

    names = [s[0] for s in STEPS]
    by_name = {s[0]: s for s in STEPS}

    # 决定要跑哪些
    if args.only or args.steps:
        sel = (args.only or args.steps).split(",")
        sel = [s.strip() for s in sel if s.strip()]
        unknown = [s for s in sel if s not in by_name]
        if unknown:
            print(f"[错误] 未知步骤 {unknown}；可用：{names}", file=sys.stderr)
            return 2
        todo = [by_name[s] for s in sel]
    elif args.from_step:
        if args.from_step not in by_name:
            print(f"[错误] 未知步骤 {args.from_step}；可用：{names}", file=sys.stderr)
            return 2
        todo = STEPS[names.index(args.from_step):]
    else:
        todo = STEPS

    # 状态
    manifest = load_manifest()
    print("=" * 74)
    print("流水线步骤状态")
    print("=" * 74)
    for name, desc, fn, skippable in STEPS:
        outputs = _outputs(name, args)
        done = bool(outputs) and exists(*outputs)
        mark = "已完成" if done else ("无产物" if not outputs else "待运行")
        extra = ""
        if name == "predict":
            extra = f"  ({count(args.masks_raw + '/*.png')} 张掩码)"
        elif name == "filter":
            extra = f"  ({count(args.masks_clean + '/*.png')} 张)"
        elif name == "regions":
            extra = f"  ({count(args.annotations_out + '/regions/*.json')} 个 regions)"
        print(f"  {name:<9} {desc:<22} {mark}{extra}")
    print()

    if args.status:
        return 0

    # 执行
    m = manifest
    failed = None
    for name, desc, fn, skippable in todo:
        outputs = _outputs(name, args)
        done = skippable and bool(outputs) and exists(*outputs)
        if done and not args.force:
            print(f"[跳过] {name}（{desc}）：产物已存在")
            continue
        print(f"[运行] {name}（{desc}）")
        t0 = time.time()
        rc = fn(args)
        dt = time.time() - t0
        m["steps"][name] = {"rc": rc, "seconds": round(dt, 1),
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_manifest(m)
        if rc != 0:
            print(f"[失败] {name} 退出码 {rc}，中止后续步骤", file=sys.stderr)
            failed = name
            break
        print(f"[完成] {name}（{dt:.1f}s）\n")

    print("=" * 74)
    if failed:
        print(f"流水线中止于步骤 {failed}")
        return 1
    print("流水线执行完毕")
    print(f"清单：{os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
